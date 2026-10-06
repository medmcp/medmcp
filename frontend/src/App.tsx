import { useCallback, useEffect, useRef, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import { Chat } from './components/Chat'
import { CloudModelBanner } from './components/CloudModelBanner'
import { CloudModelWindow } from './components/CloudModelWindow'
import { ExternalMcpBanner } from './components/ExternalMcpBanner'
import { ExternalMcpWindow } from './components/ExternalMcpWindow'
import { ModelsWindow } from './components/ModelsWindow'
import { FileExplorer } from './components/FileExplorer'
import { StackMarketplace } from './components/StackMarketplace'
import { SettingsDrawer } from './components/SettingsDrawer'
import { UpdateWindow } from './components/UpdateWindow'
import { Viewer } from './components/Viewer'
import { WorkflowPanel } from './components/WorkflowPanel'
import { GearIcon, StoreIcon, XIcon } from './components/icons'
import { ackUpdateResult, checkForUpdate, dismissUpdate, fetchUpdate, setUpdateAutoCheck } from './api'
import type { UpdateState } from './types'

/** localStorage key holding the last active chat session id (for auto-resume). */
const ACTIVE_SESSION_KEY = 'medmcp.activeSession'
/** How often the page re-reads the release record the server keeps. */
const UPDATE_POLL_MS = 30 * 60 * 1000
/** Right after a load the outcome of an update may still be on its way (the
 *  helper finishes after the new server is up), so the first minutes poll fast. */
const UPDATE_POLL_EARLY_MS = 10 * 1000
const UPDATE_POLL_EARLY_FOR_MS = 3 * 60 * 1000

function describeUpdateResult(r: NonNullable<UpdateState['last_result']>): string {
  if (r.status === 'ok') return `Updated to ${r.to}.`
  if (r.status === 'rolled_back')
    return `The update to ${r.to} did not start; ${r.from} was put back. ${r.detail}`.trim()
  return `The update to ${r.to} failed. ${r.detail}`.trim()
}

/**
 * Four-panel workspace: explorer (top left), viewer (top right),
 * workflows (bottom left), chat (bottom right).
 */
export default function App() {
  const [openPath, setOpenPath] = useState<string | null>(null)
  // Files multi-selected in the explorer — feeds the workflow batch editor.
  const [selectedPaths, setSelectedPaths] = useState<string[]>([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsAdvanced, setSettingsAdvanced] = useState(false)
  // The external-MCP window: reached from Advanced, or straight from the warning
  // banner, which is about the thing the window manages.
  const [externalOpen, setExternalOpen] = useState(false)
  // Bumped whenever external-MCP state may have changed, so the standing
  // warning re-reads it instead of waiting for a reload.
  const [externalVersion, setExternalVersion] = useState(0)
  const notifyExternalChanged = useCallback(() => setExternalVersion((v) => v + 1), [])
  // The cloud-model window and its standing warning, wired the same way.
  const [cloudOpen, setCloudOpen] = useState(false)
  const [cloudVersion, setCloudVersion] = useState(0)
  const notifyCloudChanged = useCallback(() => setCloudVersion((v) => v + 1), [])
  // The Models window: reached from the chat header's model name and from Settings.
  const [modelsOpen, setModelsOpen] = useState(false)
  const openModels = useCallback(() => setModelsOpen(true), [])
  const [marketOpen, setMarketOpen] = useState(false)
  // The release record the server keeps (checked daily there); the header
  // notice and the Settings row both read it, the window applies it.
  const [updateState, setUpdateState] = useState<UpdateState | null>(null)
  const [updateOpen, setUpdateOpen] = useState(false)
  const loadUpdate = useCallback(() => {
    // Best-effort: an unreachable endpoint must not blank the app.
    fetchUpdate()
      .then(setUpdateState)
      .catch(() => {})
  }, [])
  useEffect(() => {
    loadUpdate()
    const started = Date.now()
    let timer = 0
    const tick = () => {
      loadUpdate()
      const early = Date.now() - started < UPDATE_POLL_EARLY_FOR_MS
      timer = window.setTimeout(tick, early ? UPDATE_POLL_EARLY_MS : UPDATE_POLL_MS)
    }
    timer = window.setTimeout(tick, UPDATE_POLL_EARLY_MS)
    return () => window.clearTimeout(timer)
  }, [loadUpdate])
  const setAutoCheck = useCallback(
    (enabled: boolean) =>
      setUpdateAutoCheck(enabled)
        .then(setUpdateState)
        .catch(() => {}),
    [],
  )
  const checkUpdate = useCallback(
    () =>
      checkForUpdate()
        .then(setUpdateState)
        .catch(() => {}),
    [],
  )
  const dismissUpdateNotice = useCallback(() => {
    const v = updateState?.latest?.version
    setUpdateOpen(false)
    if (v)
      dismissUpdate(v)
        .then(setUpdateState)
        .catch(() => {})
  }, [updateState])
  const ackUpdate = useCallback(() => {
    ackUpdateResult()
      .then(setUpdateState)
      .catch(() => {})
  }, [])
  const updateResult = updateState?.last_result ?? null
  // A successful update needs no acknowledging; the failures stay until read.
  useEffect(() => {
    if (updateResult?.status !== 'ok') return
    const t = window.setTimeout(ackUpdate, 12000)
    return () => window.clearTimeout(t)
  }, [updateResult, ackUpdate])
  const updateNotice =
    updateState?.available && !updateState.dismissed ? updateState.latest : null
  // The vibe session that received the last prompt — what "Save chat as
  // workflow" distills. Survives a reconnect (which starts an empty session).
  const [distillSessionId, setDistillSessionId] = useState<string | null>(null)
  // Bumped whenever something may have written to the workspace (agent tool
  // calls, replay steps) so the explorer/viewer reload their file tree.
  const [fsVersion, setFsVersion] = useState(0)
  const notifyFsChanged = useCallback(() => setFsVersion((v) => v + 1), [])
  // True while a separator is being dragged. The viewer is the only WebGL panel;
  // its canvas is a large GPU layer that's expensive to composite every frame, so
  // we drop it (via the `is-resizing` class) for the whole drag and let it redraw
  // once on release.
  //
  // The flag must stay true for the ENTIRE pointer drag. We can't clear it on the
  // library's `onLayoutChanged`: that fires on layout "settles" mid-drag (e.g.
  // when you pause), which would un-hide the canvas and make it lag again — the
  // exact "fine, then janky" symptom. So we set it on `onLayoutChange` (per move)
  // and clear it on the real pointer release, with a timed fallback for keyboard
  // resizes that have no pointerup.
  const [resizing, setResizing] = useState(false)
  const resizeEndTimer = useRef<number | null>(null)
  const startResize = useCallback(() => {
    setResizing(true)
    if (resizeEndTimer.current != null) clearTimeout(resizeEndTimer.current)
    resizeEndTimer.current = window.setTimeout(() => setResizing(false), 400)
  }, [])
  useEffect(() => {
    const end = () => {
      if (resizeEndTimer.current != null) {
        clearTimeout(resizeEndTimer.current)
        resizeEndTimer.current = null
      }
      setResizing(false)
    }
    window.addEventListener('pointerup', end, true)
    window.addEventListener('pointercancel', end, true)
    return () => {
      window.removeEventListener('pointerup', end, true)
      window.removeEventListener('pointercancel', end, true)
    }
  }, [])
  // Chat session continuity: on load resume the last session (auto), and let
  // the user start a fresh one. `resumeId` is what the next Chat mount should
  // resume (stored id, or null for new); bumping `chatKey` remounts <Chat/> so
  // it drops its socket and reconnects with that target. Chat captures the
  // resume id at mount, so updating it here only takes effect on the next mount.
  const [resumeId, setResumeId] = useState<string | null>(() =>
    localStorage.getItem(ACTIVE_SESSION_KEY),
  )
  const [chatKey, setChatKey] = useState(0)
  const handleSessionEstablished = useCallback((id: string) => {
    localStorage.setItem(ACTIVE_SESSION_KEY, id)
    setResumeId(id)
  }, [])
  const startNewChat = useCallback(() => {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
    setResumeId(null)
    setChatKey((k) => k + 1)
  }, [])
  const openSession = useCallback((id: string) => {
    setResumeId(id)
    setChatKey((k) => k + 1)
  }, [])

  return (
    <div className={`app-shell${resizing ? ' is-resizing' : ''}`}>
      <header className="app-header">
        <span className="app-logo">MedMCP</span>
        <span className="app-subtitle">workspace</span>
        <span className="app-header-right">
          {updateNotice && (
            <button
              className="update-pill"
              title="A newer MedMCP release is available"
              onClick={() => setUpdateOpen(true)}
            >
              <span className="update-pill-dot" />v{updateNotice.version} available
            </button>
          )}
          <button className="btn-icon" title="Tool stacks" onClick={() => setMarketOpen(true)}>
            <StoreIcon />
          </button>
          <button className="btn-icon" title="Settings" onClick={() => setSettingsOpen(true)}>
            <GearIcon />
          </button>
        </span>
      </header>
      <CloudModelBanner refreshSignal={cloudVersion} onReview={() => setCloudOpen(true)} />
      <ExternalMcpBanner
        refreshSignal={externalVersion}
        onReview={() => setExternalOpen(true)}
      />
      <SettingsDrawer
        open={settingsOpen}
        onClose={() => {
          setSettingsOpen(false)
          setSettingsAdvanced(false)
        }}
        advancedOpen={settingsAdvanced}
        onAdvancedToggle={setSettingsAdvanced}
        onManageExternal={() => setExternalOpen(true)}
        externalVersion={externalVersion}
        onManageCloud={() => setCloudOpen(true)}
        onManageModels={openModels}
        cloudVersion={cloudVersion}
        update={updateState}
        onCheckUpdate={checkUpdate}
        onOpenUpdate={() => setUpdateOpen(true)}
        onSetAutoCheck={setAutoCheck}
      />
      <UpdateWindow
        open={updateOpen}
        onClose={() => setUpdateOpen(false)}
        state={updateState}
        onDismiss={dismissUpdateNotice}
      />
      {updateResult && (
        <div className={`app-toast${updateResult.status === 'ok' ? '' : ' app-toast-error'}`} role="status">
          <span>{describeUpdateResult(updateResult)}</span>
          <button className="btn-icon" title="Dismiss" onClick={ackUpdate}>
            <XIcon />
          </button>
        </div>
      )}
      <ExternalMcpWindow
        open={externalOpen}
        onClose={() => setExternalOpen(false)}
        onChanged={notifyExternalChanged}
      />
      <CloudModelWindow
        open={cloudOpen}
        onClose={() => setCloudOpen(false)}
        onChanged={notifyCloudChanged}
      />
      <ModelsWindow open={modelsOpen} onClose={() => setModelsOpen(false)} />
      <StackMarketplace open={marketOpen} onClose={() => setMarketOpen(false)} />
      <Group
        orientation="vertical"
        className="app-main"
        resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
        onLayoutChange={startResize}
      >
        <Panel defaultSize="60%" minSize="20%">
          <Group
            orientation="horizontal"
            resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}
            onLayoutChange={startResize}
          >
            <Panel defaultSize="25%" minSize="12%">
              <FileExplorer
                onOpenFile={setOpenPath}
                refreshSignal={fsVersion}
                onSelectionChange={setSelectedPaths}
                isResizing={resizing}
              />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Viewer path={openPath} isResizing={resizing} />
            </Panel>
          </Group>
        </Panel>
        <Separator className="sep sep-h" />
        <Panel minSize="15%">
          <Group orientation="horizontal" resizeTargetMinimumSize={{ fine: 8, coarse: 24 }}>
            <Panel defaultSize="25%" minSize="12%">
              <WorkflowPanel
                distillSessionId={distillSessionId}
                onWorkspaceChanged={notifyFsChanged}
                onOpenFile={setOpenPath}
                selectedPaths={selectedPaths}
              />
            </Panel>
            <Separator className="sep sep-v" />
            <Panel minSize="30%">
              <Chat
                key={chatKey}
                onPromptedSession={setDistillSessionId}
                viewedPath={openPath}
                onToolActivity={notifyFsChanged}
                resumeSessionId={resumeId}
                onSessionEstablished={handleSessionEstablished}
                onNewChat={startNewChat}
                onOpenModels={openModels}
                currentSessionId={resumeId}
                onSelectSession={openSession}
              />
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  )
}
